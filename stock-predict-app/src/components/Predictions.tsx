interface Props {
  predictions: {
    tomorrow: number;
    oneMonth: number;
  };
}

function Predictions({ predictions }: Props) {
  return (
    <div className="mt-8 space-y-2">
      <h2 className="font-bold">📅 Tahminler</h2>
      <div>Yarın: ₺ {predictions.tomorrow.toFixed(2)}</div>
      <div>1 Ay Sonra: ₺ {predictions.oneMonth.toFixed(2)}</div>
    </div>
  );
}

export default Predictions;
